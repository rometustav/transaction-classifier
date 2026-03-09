"""
This script generates a dataset of financial transaction purpose texts and corresponding transaction type labels.

Generation approach:
    - Each category has realistic text templates, in addition to shared templates.
    - Merchants are category-specific.
    - Some text is mutated to have realistic formatting differences.
    - Transaction text is generated with random variations to prevent the models from learning overly simple patterns.
    - A small amount of noise is added to demonstrate data preprocessing.

The generated dataset is saved as a CSV file with two columns:
    purpose_text       : short description of the transaction
    transaction_type   : category label to predict

Usage:
    python data/data_generation.py

Output:
    data/transactions.csv
"""

import random
import pandas as pd
from pathlib import Path

random.seed(123) # To have the same output

# Categories whose transactions often include a month reference
MONTH_RELEVANT_CATEGORIES = {"rent", "utilities", "subscription", "salary"}

shared_templates = [
    "card payment",
    "bank transfer",
    "invoice payment",
    "service payment"
]

templates = {
    "rent": [
            "rent payment",
            "apartment rent",
            "monthly rent",
            "rental payment"],
    "groceries": [
            "grocery store purchase",
            "supermarket payment",
            "grocery payment"],
    "utilities": [
            "utility bill",
            "utility payment"],
    "subscription": [
            "monthly subscription payment",
            "subscription payment",
            "recurring subscription"],
    "transport": [
            "ride payment",
            "ride invoice",
            "travel payment",
            "travel invoice"],
    "salary": [
            "monthly salary",
            "salary payment",
            "wage transfer",],
    "restaurant": [
            "restaurant payment",
            "dinner bill",
            "cafe purchase",
            "lunch payment",
            "food delivery order"],
    "shopping": [
            "online purchase",
            "online shopping",
            "online order",
            "order payment"],
    "healthcare": [
            "pharmacy purchase",
            "medical payment",
            "doctor bill"],
    "entertainment": [
            "movie ticket",
            "cinema payment",
            "concert ticket",
            "game purchase"]
}

merchants = {
    "rent": ["Landlord", "Rental Agency", "Property Management"],
    "groceries": ["Lidl", "Coop", "Selver", "Maxima"],
    "utilities": ["Enefit", "Elektrilevi", "Telia", "Elisa"],
    "subscription": ["Amazon", "Apple", "Telia", "Netflix", "Spotify", "YouTube", "Microsoft"],
    "transport": ["Bolt", "Uber", "Forus Taxi"],
    "salary": ["Employer", "Payroll", "Company Ltd"],
    "restaurant": ["McDonalds", "Hesburger", "Vapiano", "Joyce"],
    "shopping": ["Amazon", "Apple", "Zalando", "Euronics", "Arvutitark"],
    "healthcare": ["Apotheka", "Benu", "Confido", "Synlab"],
    "entertainment": ["Steam", "PlayStation", "Spotify", "Netflix", "Apollo Cinema"]
}

months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

def mutate_text(text: str) -> str:
    candidates = [text]

    candidates.append(text.lower())
    candidates.append(text.upper())

    if "payment" in text:
        candidates.append(text.replace("payment", "pay"))
    if "invoice" in text:
        candidates.append(text.replace("invoice", "inv"))
    if "subscription" in text:
        candidates.append(text.replace("subscription", "sub"))

    return random.choice(candidates)


def generate_transaction_text(label: str) -> str:
    tokens = []
    merchant_added = False
    month_added = False

    # Add merchant most of the time
    if random.random() < 0.8:
        tokens.append(random.choice(merchants[label]))
        merchant_added = True

    # Add month sometimes for relevant categories
    if random.random() < 0.35 and label in MONTH_RELEVANT_CATEGORIES:
        tokens.append(random.choice(months))
        month_added = True

    # Choose template type
    use_shared = random.random() < 0.35

    if use_shared:
        tokens.append(random.choice(shared_templates))
    else:
        tokens.append(random.choice(templates[label]))

    # If text would be too weak, add merchant clue
    if use_shared and not merchant_added and not month_added:
        tokens.append(random.choice(merchants[label]))

    # Sometimes add an extra class-specific template
    if random.random() < 0.2:
        tokens.append(random.choice(templates[label]))

    # Sometimes shuffle token order
    if random.random() < 0.15:
        random.shuffle(tokens)

    # Remove duplicate consecutive words
    final_text = []
    last_word = None
    for part in tokens:
        for word in part.split():
            if word.lower() == last_word:
                continue
            final_text.append(word)
            last_word = word.lower()

    text = " ".join(final_text).strip()
    return mutate_text(text)


def generate_dataset(samples_per_class: int = 150) -> pd.DataFrame:
    rows = []

    for label in templates:
        for _ in range(samples_per_class):
            rows.append({
                "purpose_text": generate_transaction_text(label),
                "transaction_type": label
            })

    # Small intentional noise set for preprocessing demonstration
    noise_rows = [
        {"purpose_text": "", "transaction_type": "rent"},
        {"purpose_text": None, "transaction_type": "groceries"},
        {"purpose_text": "123456", "transaction_type": "shopping"},
        {"purpose_text": "???", "transaction_type": "utilities"},
    ]

    rows.extend(noise_rows)

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def main():
    output_path = Path("data/transactions.csv")

    df = generate_dataset(samples_per_class=150)
    df.to_csv(output_path, index=False)

    print(f"Dataset generated successfully: {output_path}")
    print(f"Total rows: {len(df)}")


if __name__ == "__main__":
    main()