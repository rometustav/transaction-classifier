# Data Processing

## Preprocessing Goals

The main goals of preprocessing are:

- handle missing or invalid data
- normalize text into a consistent format
- reduce noise caused by punctuation, casing, and spacing
- prepare the data for text vectorization and model training

## 1. Handling missing and irrelevant data

The raw generated dataset intentionally includes a small amount of noisy data to demonstrate preprocessing.

Examples of such noise include:

- missing text values (`None`)
- empty strings (`""`)
- non-informative values such as `"???"`
- purely numeric values such as `"123456"`

### How missing labels are handled

Rows with missing `transaction_type` values are removed, because supervised machine learning requires a valid target label for each training sample.

### How missing text is handled

Missing `purpose_text` values are first replaced with empty strings so they can be processed consistently by the cleaning function.

### How irrelevant text is handled

After cleaning, rows whose `purpose_text` becomes empty are removed from the dataset.

This ensures that the model is trained only on samples with meaningful text content.

## 2. Text cleaning and normalization

The `purpose_text` field is normalized using a preprocessing function.

### Step 1: Convert text to lowercase

All text is converted to lowercase.

Example:

- `TELIA UTILITY BILL` → `telia utility bill`

This reduces unnecessary variation caused by capitalization.

### Step 2: Remove unwanted characters

Non-alphanumeric characters are removed using a regular expression.

Characters such as punctuation and symbols are replaced with spaces.

Example:

- `Netflix subscription payment!!!` → `Netflix subscription payment`

### Step 3: Normalize whitespace

Multiple spaces are replaced with a single space, and leading/trailing whitespace is removed.

Example:

- `"  netflix    payment   "` → `"netflix payment"`

## 3. Cleaning logic used

The preprocessing logic is implemented in `src/preprocessing.py`.

The text cleaning function performs the following operations:

1. Convert non-string values to empty strings
2. Convert text to lowercase
3. Remove unwanted characters with regex
4. Replace repeated whitespace with a single space
5. Remove leading and trailing whitespace

Example regex operations used:

- `[^a-z0-9\s]` → removes characters that are not letters, numbers, or whitespace
- `\s+` → collapses multiple whitespace characters into a single space

## 4. Why tokenization is not done manually

The project does not perform manual tokenization during preprocessing.

Instead, tokenization is handled later by the `TfidfVectorizer` during feature extraction.

This is appropriate for a classical machine learning pipeline, because:

- the text has already been normalized
- `TfidfVectorizer` automatically splits text into tokens

## 5. Output of preprocessing

The cleaned dataset is saved as:

`data/transactions_cleaned.csv`

This cleaned dataset is then used in the model training pipeline.