# Projet Training 2024 - Ingénierie des langues 

import numpy as np
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch

# Vérification de la disponibilité de CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de {device} pour l'entraînement.")

# Chemin vers le fichier JSON
path_to_json = "merged_movies.json"
dataset = load_dataset('json', data_files=path_to_json)

# Mapper les étiquettes d'âge
age_labels = {
    "Tout public": 0,
    "Tout public avec avertissement": 1,
    "Interdit - 12 ans avec avertissement": 2,
    "Interdit - 12 ans": 3,
    "Interdit - 16 ans avec avertissement": 4,
    "Interdit - 16 ans": 5,
    "Interdit - 18 ans": 6,
    "Pornographique / Film classé X": 7
}

def label_to_index(example):
    # Gestion des âges non reconnus pour éviter les crashs
    example['age'] = age_labels.get(example['age'], 0) 
    return example

dataset = dataset.map(label_to_index)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_function(examples):
    return tokenizer(examples['synopsis'], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.map(lambda examples: {'labels': examples['age']}, batched=True)

# AMÉLIORATION : Division dynamique train/test (80% / 20%) au lieu du range en dur
split_datasets = tokenized_datasets["train"].train_test_split(test_size=0.2, seed=42)
train_dataset = split_datasets["train"]
test_dataset = split_datasets["test"]

print(f"Taille de l'entraînement : {len(train_dataset)} | Taille du test : {len(test_dataset)}")

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(age_labels))
model.to(device)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    weight_decay=0.02,
    load_best_model_at_end=True,
    logging_dir='./logs'
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

print("Début de l'entraînement...")
trainer.train()

model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')
print("Modèle sauvegardé avec succès.")

# Nettoyage de la mémoire GPU
if torch.cuda.is_available():
    torch.cuda.empty_cache()

def predict(synopsis):
    model.eval()
    inputs = tokenizer(synopsis, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    return predictions.item()

# Test du modèle
test_synopsis = "Une jeune infirmière soupçonne sa famille d'être possédée par le Diable après qu'elle ait accepté d'écrire l'histoire glaçante d'un de ses patients, hanté par une entité démoniaque."
age_prediction = predict(test_synopsis)
# Retrouver le label textuel à partir de l'index
predicted_label = list(age_labels.keys())[list(age_labels.values()).index(age_prediction)]
print(f"Synopsis de test : {test_synopsis}")
print(f"Age conseillé prédit : {age_prediction} ({predicted_label})")