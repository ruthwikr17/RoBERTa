import pandas as pd
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import torch
import torch.nn as nn

# Load dataset from CSV
df = pd.read_csv("Data/sarcasm_dataset.csv")
df = df.rename(columns=str.lower)

# Extract headlines and labels
headlines = df["text"].tolist()
labels = df["label"].tolist()

# Split data into training and testing sets (80% train, 20% test)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    headlines, labels, test_size=0.2, random_state=42
)

class_weights = compute_class_weight(
    class_weight="balanced", classes=np.unique(train_labels), y=train_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# Load BERT tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# Tokenize the data
train_encodings = tokenizer(
    train_texts, truncation=True, padding=True, return_tensors="pt", max_length=128
)
test_encodings = tokenizer(
    test_texts, truncation=True, padding=True, return_tensors="pt", max_length=128
)


# Create a custom dataset class
class SarcasmDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = SarcasmDataset(train_encodings, train_labels)
test_dataset = SarcasmDataset(test_encodings, test_labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    evaluation_strategy="steps",
    eval_steps=100,
    logging_dir="./logs",
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    learning_rate=2e-5,
)


# Define compute_metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(model.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# Initialize Trainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)
preds = trainer.predict(test_dataset).predictions.argmax(-1)
print("Confusion Matrix:")
print(confusion_matrix(test_labels, preds))


model.save_pretrained("roberta_sarcasm_model")
tokenizer.save_pretrained("roberta_sarcasm_model")

print("Model saved")
