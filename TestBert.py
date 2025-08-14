import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert_sarcasm_model")
model = BertForSequenceClassification.from_pretrained("bert_sarcasm_model")
model.eval()

def predict_sarcasm(tweet):
    inputs = tokenizer(tweet, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
        predicted_class = np.argmax(probs)
        label = "Sarcastic" if predicted_class == 1 else "Not Sarcastic"
        print(f"Tweet: {tweet}")
        print(f"Prediction: {label}")
        print(f"Confidence: {probs[predicted_class]:.4f}")

# Load dataset
df = pd.read_csv("Data/test_dataset 3.csv")
texts = df["tweets"].tolist()
labels = df["class"].tolist()


# Split
_, test_texts, _, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)


# Tokenize test texts
encodings = tokenizer(
    test_texts, truncation=True, padding=True, max_length=128, return_tensors="pt"
)


# Move to same device as model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
encodings = {key: val.to(device) for key, val in encodings.items()}


# Predict
with torch.no_grad():
    outputs = model(**encodings)
    preds = torch.argmax(outputs.logits, axis=1).cpu().numpy()


# Evaluate
print("Accuracy:", accuracy_score(test_labels, preds))
print("Classification Report:\n", classification_report(test_labels, preds))


# Confusion Matrix
cm = confusion_matrix(test_labels, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Not Sarcastic", "Sarcastic"],
    yticklabels=["Not Sarcastic", "Sarcastic"],
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


# Metric Bar Chart
precision = precision_score(test_labels, preds)
recall = recall_score(test_labels, preds)
f1 = f1_score(test_labels, preds)
accuracy = accuracy_score(test_labels, preds)

metrics = [accuracy, precision, recall, f1]
labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

plt.figure(figsize=(8, 6))
sns.barplot(x=labels, y=metrics, palette="viridis")
plt.ylim(0, 1)
plt.title("BERT Model Performance Metrics")
plt.ylabel("Score")
plt.show()

if __name__ == "__main__":
    tweet_input = input("Enter a tweet to check for sarcasm: ")
    predict_sarcasm(tweet_input)
