import pandas as pd
import numpy as np
import torch
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    GPT2Tokenizer,
    GPT2ForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# Load and preprocess the dataset
def load_and_preprocess_data(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath)

    # Check required columns
    required_cols = {"tweets", "class"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain {required_cols}")

    # Basic preprocessing
    df["tweets"] = df["tweets"].str.lower()
    df["tweets"] = df["tweets"].str.replace(r"http\S+|www\S+|https\S+", "", regex=True)
    df["tweets"] = df["tweets"].str.replace(r"[^\w\s#@]", "", regex=True)

    # Split into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    return train_df, test_df


# Metrics computation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


# Model training function
def train_model(model_name, train_dataset, test_dataset):
    # Select appropriate tokenizer and model
    if model_name == "bert":
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )
    elif model_name == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=2
        )
    elif model_name == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token  # Set padding token
        model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2)
        model.config.pad_token_id = tokenizer.pad_token_id

    # Tokenize the datasets
    def Tokenize(batch):
        return tokenizer(
            batch["tweets"], padding="max_length", truncation=True, max_length=128
        )

    train_dataset = train_dataset.map(Tokenize, batched=True)
    test_dataset = test_dataset.map(Tokenize, batched=True)

    # Set format for PyTorch
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./results_{model_name}",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate on test set
    metrics = trainer.evaluate()
    # Remove "eval_" prefix from metric names
    cleaned_metrics = {k.replace("eval_", ""): v for k, v in metrics.items()}
    print(f"{model_name.upper()} Metrics: {cleaned_metrics}")

    return cleaned_metrics


# Main execution
def main():
    # Load and preprocess data
    train_df, _ = load_and_preprocess_data("Data/sarcasm_dataset.csv")
    _, test_df = load_and_preprocess_data("Data/test_dataset 2.csv")

    train_df = train_df.rename(columns={"class": "labels"})
    test_df = test_df.rename(columns={"class": "labels"})

    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Dictionary to store results
    results = {}

    # Train and evaluate each model
    for model_name in ["bert", "roberta", "gpt2"]:
        print(f"\nTraining {model_name.upper()} model...")
        metrics = train_model(model_name, train_dataset, test_dataset)
        results[model_name] = metrics

    # Compare results
    print("\nModel Comparison:")
    comparison_df = pd.DataFrame.from_dict(results, orient="index")
    print(comparison_df)

    # Visualize results
    plot_results(comparison_df)

    # Determine best model
    best_model = comparison_df["f1"].idxmax()
    print(
        f"\nBest performing model: {best_model.upper()} with F1-score: {comparison_df.loc[best_model]['f1']:.4f}"
    )


# Visualization function
def plot_results(comparison_df):
    metrics = ["accuracy", "precision", "recall", "f1"]
    plt.figure(figsize=(12, 6))

    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i + 1)
        sns.barplot(x=comparison_df.index, y=comparison_df[metric])
        plt.title(metric.capitalize())
        plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.show()


if __name__ == "__main__":
    main()
