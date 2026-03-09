# How an LLM or Transformer model could be used for this classification task

## 1. Problem

This task is a **multi-class text classification problem**.

Input example:

`"Lidl grocery store purchase"`

Target label:

`"groceries"`

For a transformer-based approach, each transaction text would be tokenized and passed into a pre-trained language model.  
A classification head on top of the model would then output probabilities for all possible classes, such as:

- rent
- groceries
- utilities
- subscription
- transport
- salary
- restaurant
- shopping
- entertainment
- healthcare

The predicted class would be the one with the highest probability.

## 2. Suitable models

A practical model choice for this task would be a small or medium pre-trained transformer such as:

- `distilbert-base-uncased`
- `bert-base-uncased`

## 3. Fine-tuning approach

### Step 1: Prepare the dataset
The dataset would need two columns:

- `purpose_text`
- `transaction_type`

The labels would then be converted into integer IDs.

Example:

- rent -> 0
- groceries -> 1
- utilities -> 2
- ...

### Step 2: Tokenize the text
A tokenizer matching the chosen model would convert raw text into token IDs.

For example:

`"Netflix monthly subscription payment"`

would be tokenized into subword tokens and converted into model input tensors.

### Step 3: Add a classification head
A pre-trained transformer model would be loaded with a sequence classification head sized to the number of classes.

For example, if there are 10 classes, the output layer would have 10 neurons.

### Step 4: Fine-tune on labeled data
The model would then be trained on the transaction dataset using supervised learning.

Typical settings would include:

- train/validation split
- cross-entropy loss
- AdamW optimizer
- small learning rate
- 2 to 5 epochs
- batch size based on available hardware

### Step 5: Evaluate performance
The transformer model would be evaluated using the same metrics as the classical models:

- accuracy
- precision
- recall
- F1-score

This would allow direct comparison between the LLM/transformer approach and the classical TF-IDF models.
